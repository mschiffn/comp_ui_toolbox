%
% superclass for all grids
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-03-22
%
classdef grid

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2	% number of dimensions (1)
        %TODO: positions, N_points

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid( N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) create regular grids
            %--------------------------------------------------------------
            objects = repmat( objects, size( N_dimensions ) );

            % check and set independent properties
            for index_object = 1:numel( N_dimensions )

                % set independent properties
                objects( index_object ).N_dimensions = N_dimensions( index_object );

            end % for index_object = 1:numel( N_dimensions )

        end % function objects = grid( N_dimensions )

    end % methods

end % classdef grid
