%
% superclass for all transducer control settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-05
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active ( 1, : ) double { mustBeInteger, mustBeFinite }	% indices of active array elements (1)
        impulse_responses ( 1, : ) discretizations.signal_matrix         % impulse responses of active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting( indices_active, impulse_responses )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % ensure cell array for impulse_responses
            if ~iscell( impulse_responses )
                impulse_responses = { impulse_responses };
            end

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

                    case 'discretizations.signal'

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

                    case 'discretizations.signal_matrix'

                        % ensure single signal matrix of correct size
                        if ~isscalar( impulse_responses{ index_object } ) || ( numel( indices_active{ index_object } ) ~= impulse_responses{ index_object }.N_signals )
                            errorStruct.message     = sprintf( 'impulse_responses{ %d } must be a scalar and contain %d signals!', index_object, numel( indices_active{ index_object } ) );
                            errorStruct.identifier	= 'setting:SizeMismatch';
                            error( errorStruct );
                        end

                end % switch class( impulse_responses{ index_object } )

                % set independent properties
                objects( index_object ).indices_active = indices_active{ index_object };
                objects( index_object ).impulse_responses = impulse_responses{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = setting( indices_active, impulse_responses )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function settings = discretize( settings, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure correct number of arguments
            if nargin ~= 3
                errorStruct.message     = 'Three arguments are required!';
                errorStruct.identifier	= 'discretize:NumberArguments';
                error( errorStruct );
            end

            % multiple settings / single intervals_t
            if ~isscalar( settings ) && isscalar( intervals_t )
                intervals_t = repmat( intervals_t, size( settings ) );
            end

            % multiple settings / single intervals_f
            if ~isscalar( settings ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings, intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute Fourier transform samples
            %--------------------------------------------------------------
            % iterate transducer control settings
            for index_object = 1:numel( settings )

                % compute Fourier transform samples
                settings( index_object ).impulse_responses = fourier_transform( settings( index_object ).impulse_responses, intervals_t( index_object ), intervals_f( index_object ) );

                % merge transforms to ensure class signal_matrix
                settings( index_object ).impulse_responses = merge( settings( index_object ).impulse_responses );

            end % for index_object = 1:numel( settings )

        end % function settings = discretize( settings, intervals_t, intervals_f )

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

end % classdef setting
