%
% superclass for all transducer control settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-04-02
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active ( 1, : ) double { mustBeInteger, mustBeFinite }	% indices of active array elements (1)
        impulse_responses ( 1, : ) discretizations.signal_matrix        % impulse responses of active channels

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

                            samples = repmat( impulse_responses{ index_object }.samples, [ numel( indices_active{ index_object } ), 1 ] );
                            impulse_responses{ index_object } = discretizations.signal_matrix( impulse_responses{ index_object }.axis, samples );

                        else

                            % ensure equal number of dimensions and sizes of cell array contents
                            auxiliary.mustBeEqualSize( indices_active{ index_object }, impulse_responses{ index_object } );

                            % assemble signal_matrix for equal axes
                            if isequal( impulse_responses{ index_object }.axis )
                                samples = reshape( impulse_responses{ index_object }.samples, [ numel( impulse_responses{ index_object } ), abs( impulse_responses{ index_object }(1).axis ) ] );
                                impulse_responses{ index_object } = discretizations.signal_matrix( impulse_responses{ index_object }.axis(1), samples );
                            end

                        end

                    case 'discretizations.signal_matrix'

                        % ensure single signal matrix of correct size
                        if ~isscalar( impulse_responses{ index_object } ) || ( numel( indices_active{ index_object } ) ~= impulse_responses{ index_object }.N_signals )
                            errorStruct.message     = sprintf( 'impulse_responses{ %d } must be a scalar and contain %d signals!', index_object, numel( indices_active{ index_object } ) );
                            errorStruct.identifier	= 'setting:SizeMismatch';
                            error( errorStruct );
                        end

                    otherwise

                        errorStruct.message     = sprintf( 'impulse_responses{ %d } has to be discretizations.signal_matrix!', index_object );
                        errorStruct.identifier	= 'setting:NoSignalMatrix';
                        error( errorStruct );

                end % switch class( impulse_responses{ index_object } )

%                 % ensure row vectors
%                 if ~isrow( indices_active{ index_object } )
%                     errorStruct.message     = sprintf( 'The contents of indices_active{ %d } and impulse_responses{ %d } must be row vectors!', index_object, index_object );
%                     errorStruct.identifier	= 'setting:SizeMismatch';
%                     error( errorStruct );
%                 end

                % set independent properties
                objects( index_object ).indices_active = indices_active{ index_object };
                objects( index_object ).impulse_responses = impulse_responses{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = setting( indices_active, impulse_responses )

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
