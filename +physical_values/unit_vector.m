%
% superclass for all unit vectors
%
% author: Martin F. Schiffner
% date: 2019-01-30
% modified: 2019-01-30
%
% TODO: use cell arrays as arguments
classdef unit_vector

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        components ( 1, : ) double { mustBeReal, mustBeFinite, mustBeNonempty } = [0, 1]	% unit vector

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = unit_vector( components )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                return;
            end

            % ensure matrix argument
            if ~ismatrix( components ) || ~isreal( components ) || ~all( isfinite( components(:) ) )
                errorStruct.message     = 'components must be a real-valued finite matrix!';
                errorStruct.identifier	= 'unit_vector:NoRealFiniteMatrix';
                error( errorStruct );
            end

            % ensure l2-norms of unity
            norms = sqrt( sum( abs( components ).^2, 2 ) );
            if ~all( abs( norms - 1 ) < eps )
                errorStruct.message     = 'Rows of the argument must be unit vectors!';
                errorStruct.identifier	= 'unit_vector:NoRealMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create unit vectors
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = size( components, 1 );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects
                objects( index_object ).components = components( index_object, : );
            end

        end % function objects = unit_vector( components )

	end % methods

end % classdef unit_vector
