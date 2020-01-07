%
% superclass for all sequence reweighting options
%
% author: Martin F. Schiffner
% date: 2019-09-17
% modified: 2020-01-03
%
classdef reweighting_sequence < regularization.options.reweighting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        q ( 1, 1 ) double { mustBeNonnegative, mustBeLessThanOrEqual( q, 2 ) } = 0.5	% norm parameter
        epsilon_n ( :, 1 ) double { mustBeNonnegative } = 1 ./ ( 1 + (1:5) )            % reweighting sequence

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = reweighting_sequence( q, epsilon_n )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid q

            % ensure cell array for epsilon_n
            if ~iscell( epsilon_n )
                epsilon_n = { epsilon_n };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( q, epsilon_n );

            %--------------------------------------------------------------
            % 2.) create sequence reweighting options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.reweighting( size( q ) );

            % iterate sequence reweighting options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).q = q( index_object );
                objects( index_object ).epsilon_n = epsilon_n{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = reweighting_sequence( epsilon_n )

	end % methods

end % classdef reweighting_sequence < regularization.options.reweighting
