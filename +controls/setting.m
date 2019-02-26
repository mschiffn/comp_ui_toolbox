%
% superclass for all transducer control settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-02-26
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active ( 1, : ) double { mustBeInteger, mustBeFinite }      % indices of active array elements (1)
        impulse_responses ( 1, : ) physical_values.impulse_response         % impulse responses of active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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
            % construct objects
            objects = repmat( objects, size( indices_active ) );

            % set independent properties
            for index_object = 1:numel( indices_active )

                % ensure equal number of dimensions and sizes of cell array contents
                auxiliary.mustBeEqualSize( indices_active{ index_object }, impulse_responses{ index_object } );
                % assertion: all cell array contents have equal sizes

                % ensure row vectors
                if ~isrow( indices_active{ index_object } )
                    errorStruct.message     = sprintf( 'The contents of indices_active{ %d } and impulse_responses{ %d } must be row vectors!', index_object, index_object );
                    errorStruct.identifier	= 'setting:SizeMismatch';
                    error( errorStruct );
                end

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
