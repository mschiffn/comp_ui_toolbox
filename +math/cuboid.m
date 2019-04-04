%
% superclass for all d-cuboids
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-04-01
%
classdef cuboid < math.parallelotope

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = cuboid( edge_lengths )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for edge_lengths
            if ~iscell( edge_lengths )
                edge_lengths = { edge_lengths };
            end

            %--------------------------------------------------------------
            % 2.) create canonical bases of adequate dimensions
            %--------------------------------------------------------------
            basis = cell( size( edge_lengths ) );
            for index_object = 1:numel( edge_lengths )
                basis{ index_object } = math.unit_vector( eye( numel( edge_lengths{ index_object } ) ) )';
            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.parallelotope( edge_lengths, basis );

        end % function objects = cuboid( edge_lengths )

    end % methods

end % classdef cuboid < math.parallelotope
