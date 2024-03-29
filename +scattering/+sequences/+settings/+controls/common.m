%
% superclass for all transducer control settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2020-07-14
%
classdef (Abstract) common

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active ( 1, : ) double { mustBePositive, mustBeInteger, mustBeFinite }	% indices of active array elements (1)
        impulse_responses ( 1, : ) processing.signal_matrix                             % impulse responses of active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = common( indices_active, impulse_responses )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % validate number of input arguments
            narginchk( 2, 2 );

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % property validation function ensures finite positive integers for indices_active

            % ensure cell array for impulse_responses
            if ~iscell( impulse_responses )
                impulse_responses = { impulse_responses };
            end

            % property validation function ensures class processing.signal_matrix for impulse_responses

            % ensure equal number of dimensions and sizes of cell arrays
            auxiliary.mustBeEqualSize( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 2.) create transducer control settings
            %--------------------------------------------------------------
            % repeat default transducer control setting
            objects = repmat( objects, size( indices_active ) );

            % iterate transducer control settings
            for index_object = 1:numel( indices_active )

                switch class( impulse_responses{ index_object } )

                    case 'processing.signal'

                        % multiple indices_active{ index_object } / single impulse_responses{ index_object }
                        if ~isscalar( indices_active{ index_object } ) && isscalar( impulse_responses{ index_object } )
                            impulse_responses{ index_object } = repmat( impulse_responses{ index_object }, size( indices_active{ index_object } ) );
                        end

                        % ensure equal number of dimensions and sizes of cell array contents
                        auxiliary.mustBeEqualSize( indices_active{ index_object }, impulse_responses{ index_object } );

                        % try to merge compatible signals into a single signal matrix
                        try
                            impulse_responses{ index_object } = merge( impulse_responses{ index_object } );
                        catch
                        end

                    case 'processing.signal_matrix'

                        % ensure single signal matrix of correct size
                        if ~isscalar( impulse_responses{ index_object } ) || ( numel( indices_active{ index_object } ) ~= impulse_responses{ index_object }.N_signals )
                            errorStruct.message = sprintf( 'impulse_responses{ %d } must be a scalar and contain %d signals!', index_object, numel( indices_active{ index_object } ) );
                            errorStruct.identifier = 'common:SizeMismatch';
                            error( errorStruct );
                        end

                end % switch class( impulse_responses{ index_object } )

                % set independent properties
                objects( index_object ).indices_active = indices_active{ index_object };
                objects( index_object ).impulse_responses = impulse_responses{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = common( indices_active, impulse_responses )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function settings = discretize( settings, Ts_ref, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % validate number of input arguments
            narginchk( 1, 3 );

            % ensure class scattering.sequences.settings.controls.common
            if ~isa( settings, 'scattering.sequences.settings.controls.common' )
                errorStruct.message = 'settings must be scattering.sequences.settings.controls.common!';
                errorStruct.identifier = 'discretize:NoSettings';
                error( errorStruct );
            end

            % ensure existence of Ts_ref
            if nargin < 2
                Ts_ref = [];
            end

            % method fourier_transform ensures valid Ts_ref

            % ensure existence of intervals_f
            if nargin < 3
                intervals_f = [];
            end

            % method fourier_transform ensures valid intervals_f

            % multiple settings / single Ts_ref
            if ~isscalar( settings ) && isscalar( Ts_ref )
                Ts_ref = repmat( Ts_ref, size( settings ) );
            end

            % multiple settings / single intervals_f
            if ~isscalar( settings ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings, Ts_ref, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            % iterate transducer control settings
            for index_object = 1:numel( settings )

                % compute Fourier transform samples
                settings( index_object ).impulse_responses = fourier_transform( settings( index_object ).impulse_responses, Ts_ref( index_object ), intervals_f( index_object ) );

                % merge transforms to ensure class processing.signal_matrix
                settings( index_object ).impulse_responses = merge( settings( index_object ).impulse_responses );

            end % for index_object = 1:numel( settings )

        end % function settings = discretize( settings, Ts_ref, intervals_f )

        %------------------------------------------------------------------
        % unique indices of active array elements
        %------------------------------------------------------------------
        function [ indices_unique, indices_local_to_unique ] = unique_indices_active( settings )

            %--------------------------------------------------------------
            % 1.) numbers of members and cumulative sum
            %--------------------------------------------------------------
            N_active = cellfun( @numel, { settings.indices_active } );
            N_active_cs = [ 0, cumsum( N_active ) ];

            %--------------------------------------------------------------
            % 2.) extract unique indices of active array elements
            %--------------------------------------------------------------
            [ indices_unique, ia, ic ] = unique( [ settings.indices_active ] );

            %--------------------------------------------------------------
            % 3.) map indices of active array elements in each setting to the unique indices
            %--------------------------------------------------------------
            indices_local_to_unique = cell( size( settings ) );

            for index_set = 1:numel( settings )

                index_start = N_active_cs( index_set ) + 1;
                index_stop = index_start + N_active( index_set ) - 1;

                indices_local_to_unique{ index_set } = ic( index_start:index_stop );

            end % for index_set = 1:numel( settings )

        end % function [ indices_unique, indices_local_to_unique ] = unique_indices_active( settings )

        %------------------------------------------------------------------
        % unique deltas
        %------------------------------------------------------------------
        function deltas = unique_deltas( settings )

            % extract impulse_responses
            impulse_responses = reshape( { settings.impulse_responses }, size( settings ) );

            % specify cell array for deltas
            deltas = cell( size( settings ) );

            % iterate transducer control settings
            for index_setting = 1:numel( settings )

                % ensure equal subclasses of math.sequence_increasing_regular_quantized
                auxiliary.mustBeEqualSubclasses( 'math.sequence_increasing_regular_quantized', impulse_responses{ index_setting }.axis );

                % extract regular axes
                axes = reshape( [ impulse_responses{ index_setting }.axis ], size( impulse_responses{ index_setting } ) );

                % ensure equal subclasses of physical_values.physical_quantity
                auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', axes.delta );

                % extract deltas as row vector
                deltas{ index_setting } = reshape( [ axes.delta ], size( impulse_responses{ index_setting } ) );

            end % for index_setting = 1:numel( settings )

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', deltas{ : } );

            % extract unique deltas
            deltas = unique( cat( 2, deltas{ : } ) );

        end % function deltas = unique_deltas( settings )

        %------------------------------------------------------------------
        % support
        %------------------------------------------------------------------
        function results = support( objects )

            % allocate memory for results
            t_lbs = physical_values.time( zeros( size( objects ) ) );
            t_ubs = physical_values.time( zeros( size( objects ) ) );

            for index_object = 1:numel( objects )

                t_start = zeros( 1, numel( objects( index_object ).indices_active ) );
                t_stop = zeros( 1, numel( objects( index_object ).indices_active ) );

                for index_element = 1:numel( objects( index_object ).indices_active )
                    t_start( index_element ) = impulse_responses( index_element ).set_t.S( 1 );
                    t_stop( index_element ) = impulse_responses( index_element ).set_t.S( end );
                end

            end

        end % function results = support( objects )

	end % methods

end % classdef (Abstract) common
