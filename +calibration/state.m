%
% superclass for all states
%
% author: Martin F. Schiffner
% date: 2019-06-12
% modified: 2019-06-12
%
classdef state

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        position_target ( 1, : ) physical_values.length
        c_avg ( 1, 1 ) physical_values.velocity = physical_values.meter_per_second( 1540 );

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = state( positions_target, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % ensure cell array for positions_target
            if ~iscell( positions_target )
                if ~isrow( positions_target )
                    positions_target = mat2cell( positions_target, ones( size( positions_target, 1 ), 1 ), size( positions_target, 2 ) );
                else
                    positions_target = { positions_target };
                end
            end

            % multiple positions_target / single c_avg
            if ~isscalar( positions_target ) && isscalar( c_avg )
                c_avg = repmat( c_avg, size( positions_target ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( positions_target, c_avg );

            %--------------------------------------------------------------
            % 2.) create states
            %--------------------------------------------------------------
            % repeat default state
            objects = repmat( objects, size( positions_target ) );

            % iterate states
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).position_target = positions_target{ index_object };
                objects( index_object ).c_avg = c_avg( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = state( positions_target, c_avg )

	end % methods

end % classdef state
